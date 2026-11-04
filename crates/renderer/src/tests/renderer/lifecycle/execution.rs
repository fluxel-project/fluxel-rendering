//! Scripted execution mode for the renderer witness backend.

use super::backend::*;
use super::*;

pub(super) struct WitnessMode;
pub(super) struct WitnessState {
    native: Arc<Native>,
    device: Device<WitnessBackend>,
    steps: Vec<StepId>,
    lanes: Vec<LaneId>,
    calls: usize,
    present: Option<PresentReceipt>,
}
pub(super) struct WitnessOutcome {
    guard: Arc<OutcomeGuard>,
    error: Option<Error>,
    status: ExecutionStatus,
    failed_step: Option<FailedStep>,
    accepted: Option<Vec<StepId>>,
    affected: Option<Vec<StepId>>,
    checkpoints: Option<Vec<Checkpoint<WitnessBackend>>>,
    presents: Option<Vec<PresentReceipt>>,
}

impl ExecutionOutcome<WitnessBackend> for WitnessOutcome {
    fn facts(&self) -> ExecutionFacts<'_, WitnessBackend> {
        // The original outcome owns this guard; keep that ownership observable.
        let _ = &self.guard;
        ExecutionFacts {
            status: self.status,
            failed_step: self.failed_step,
            accepted: self.accepted.as_deref(),
            affected: self.affected.as_deref(),
            checkpoints: self.checkpoints.as_deref(),
            presents: self.presents.as_deref(),
        }
    }
    fn error(&self) -> Option<&Error> {
        self.error.as_ref()
    }
}
impl ExecutionBackend for WitnessBackend {
    type Mode = WitnessMode;
}
impl DeferredBackend for WitnessBackend {
    type NativeRecordedWork = ();
    fn finish_encoder(_: (), _: LaneId) -> Result<()> {
        Ok(())
    }
    unsafe fn submit(_: &Arc<Native>, _: SubmitPacket<Self>) -> SubmitOutcome<Self> {
        unreachable!()
    }
    unsafe fn resume(_: &Arc<Native>, _: DeferredTail<Self>) -> SubmitOutcome<Self> {
        unreachable!()
    }
}
impl ExecutionMode<WitnessBackend> for WitnessMode {
    type State = WitnessState;
    type Outcome = WitnessOutcome;
    unsafe fn begin(
        device: &Device<WitnessBackend>,
        plan: SubmissionPlan<WitnessBackend>,
        _: CompletionRequest,
    ) -> Result<Self::State> {
        if unsafe { device.native_ref() }
            .fault_enabled
            .load(Ordering::SeqCst)
            && matches!(
                unsafe { device.native_ref() }.script,
                Script::BeginDeviceLost
            )
        {
            return Err(Error::DeviceLost);
        }
        // SAFETY: the test fixture constructed this Device from the same Arc.
        Ok(WitnessState {
            native: unsafe { Arc::clone(device.native_ref()) },
            device: device.clone(),
            steps: plan
                .steps()
                .iter()
                .enumerate()
                .map(|(index, _)| plan.step(index))
                .collect(),
            lanes: plan.steps().iter().map(|step| step.lane).collect(),
            calls: 0,
            present: None,
        })
    }
    fn commands<F: FnOnce(&mut CommandEncoder<WitnessBackend>) -> Result<()>>(
        state: &mut Self::State,
        step: StepId,
        encode: F,
    ) -> Result<()> {
        let call = state.calls;
        state.calls += 1;
        state.native.command_calls.fetch_add(1, Ordering::SeqCst);
        if state.native.fault_enabled.load(Ordering::SeqCst)
            && matches!(state.native.script, Script::RecordingDeviceLost)
            && call == 1
        {
            Err(Error::DeviceLost)
        } else if matches!(state.native.script, Script::RecordingFailureAfterPrefix) && call == 1 {
            Err(Error::Backend("scripted recording failure".to_owned()))
        } else {
            if state.steps.get(call).copied() != Some(step) {
                return Err(Error::InvalidState);
            }
            let mut encoder = state.device.encoder(state.lanes[call])?;
            encode(&mut encoder)?;
            let _ = encoder.finish()?;
            Ok(())
        }
    }
    fn present(
        state: &mut Self::State,
        step: StepId,
        mut binding: PresentBinding<WitnessBackend>,
    ) -> std::result::Result<(), PresentError<WitnessBackend>> {
        match state.native.script {
            Script::SurfacePresentRejected => Err(PresentError {
                error: Error::SurfaceOutdated,
                binding: Some(binding),
            }),
            Script::SurfacePresentDeviceLost => Err(PresentError {
                error: Error::DeviceLost,
                binding: None,
            }),
            _ => {
                unsafe { binding.try_handoff() }.map_err(|error| PresentError {
                    error,
                    binding: Some(binding),
                })?;
                state.present = Some(PresentReceipt {
                    step: Some(step),
                    accepted: true,
                });
                Ok(())
            }
        }
    }
    unsafe fn finish(state: Self::State) -> Self::Outcome {
        state.native.finishes.fetch_add(1, Ordering::SeqCst);
        let guard = Arc::new(OutcomeGuard);
        *state.native.outcome.lock().unwrap() = Arc::downgrade(&guard);
        let script = if state.native.fault_enabled.load(Ordering::SeqCst) {
            state.native.script
        } else {
            Script::AcceptedExact
        };
        let failed_step = matches!(script, Script::PartialIssued).then_some(FailedStep {
            step: state.steps[1],
            effect: StepEffect::Partial,
        });
        let (status, accepted, affected, checkpoints) = match script {
            Script::CleanRejected => (
                ExecutionStatus::Rejected,
                Some(Vec::new()),
                Some(Vec::new()),
                None,
            ),
            Script::RecordingFailureAfterPrefix
            | Script::RecordingDeviceLost
            | Script::PartialAccepted => (
                ExecutionStatus::PartiallyAccepted,
                Some(vec![state.steps[0]]),
                Some(vec![state.steps[1]]),
                None,
            ),
            Script::PartialIssued => (
                ExecutionStatus::Failed,
                Some(vec![state.steps[0]]),
                Some(vec![state.steps[1]]),
                None,
            ),
            Script::Indeterminate | Script::FinishDeviceLost | Script::SurfacePresentDeviceLost => {
                (ExecutionStatus::Indeterminate, None, None, None)
            }
            Script::SurfacePresentRejected => (
                ExecutionStatus::Rejected,
                Some(Vec::new()),
                Some(Vec::new()),
                None,
            ),
            Script::AcceptedExact
            | Script::AcceptedWithError
            | Script::AcceptedWaitPending
            | Script::AcceptedWrongFrontier
            | Script::AcceptedIdleFailure
            | Script::AcceptedIdlePending
            | Script::AcceptedIdleDeviceLost
            | Script::AcceptedCompletionDeviceLost
            | Script::AcceptedReadbackDeviceLost
            | Script::AllocationDeviceLost
            | Script::MapDeviceLost
            | Script::FlushDeviceLost
            | Script::BeginDeviceLost => {
                let frontier = if matches!(state.native.script, Script::AcceptedWrongFrontier) {
                    state.steps[0]
                } else {
                    *state.steps.last().expect("copy plan has readback")
                };
                let completion_id = state.native.next_completion.fetch_add(1, Ordering::SeqCst) + 1;
                let point = unsafe {
                    CompletionPoint::from_native(
                        state.device.id(),
                        Arc::new(CompletionMarker { id: completion_id }),
                    )
                };
                (
                    ExecutionStatus::Accepted,
                    Some(state.steps),
                    Some(Vec::new()),
                    Some(vec![Checkpoint {
                        point,
                        covers: CompletionCoverage {
                            lane_frontiers: vec![(state.lanes[0], frontier)],
                        },
                    }]),
                )
            }
        };
        WitnessOutcome {
            guard,
            error: if matches!(script, Script::AcceptedWithError) {
                Some(Error::Backend("accepted with error".to_owned()))
            } else {
                matches!(
                    script,
                    Script::FinishDeviceLost | Script::SurfacePresentDeviceLost
                )
                .then_some(Error::DeviceLost)
            },
            status,
            failed_step,
            accepted,
            affected,
            checkpoints,
            presents: state.present.map(|receipt| vec![receipt]),
        }
    }
}
