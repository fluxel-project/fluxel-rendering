//! Compiled single-Graph Deferred-to-Forward capture plan.

use super::forward_capture_declaration::DeferredForwardCaptureDeclaration;
use super::*;
use crate::graph::forward::ForwardReceiverTerminal;

pub(crate) struct DeferredForwardCapturePlan<B: ResourceBackend> {
    pub(super) producer: DeferredPlan<B>,
    pub(super) producer_composition:
        Option<super::capture_producer::DeferredCaptureProducerComposition>,
    pub(super) receiver: crate::graph::forward::ForwardCaptureReceiver,
    pub(super) terminal: ForwardReceiverTerminal,
    pub(super) submission: SubmissionPlan<B>,
    pub(super) execution: CompiledExecution,
    pub(super) id: PlanId,
}

impl<B: ResourceBackend> DeferredForwardCapturePlan<B> {
    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: DeferredForwardCaptureDeclaration,
    ) -> Result<Self> {
        if id.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let execution = CompiledExecution::compile(&declaration.producer.graph)?;
        let mut expected = namespace_passes(&declaration.producer);
        if let Some(composition) = declaration.producer_composition.as_ref() {
            composition.append_passes(&mut expected);
        }
        declaration.receiver.append_passes(&mut expected);
        match declaration.terminal {
            ForwardReceiverTerminal::Offscreen { pass }
            | ForwardReceiverTerminal::Surface { pass, .. } => expected.push(pass),
            ForwardReceiverTerminal::HdrReadback(terminal) => {
                expected.extend([terminal.composite_pass, terminal.readback_pass]);
            }
        }
        if execution.logical().passes() != expected {
            return Err(Error::InvalidState);
        }
        let lane = declaration.producer.lane;
        let surface = matches!(
            declaration.terminal,
            ForwardReceiverTerminal::Surface { .. }
        );
        let submission = if surface {
            let index = u32::try_from(expected.len()).map_err(|_| Error::InvalidRange)?;
            let mut steps = execution.steps(device, id, lane)?;
            steps.push(fluxel_rhi::api::StepDesc {
                lane,
                kind: fluxel_rhi::api::StepKind::Present,
                waits: vec![fluxel_rhi::api::StepWait {
                    previous: StepId {
                        plan: id,
                        index: index.checked_sub(1).ok_or(Error::InvalidState)?,
                    },
                    destination: fluxel_rhi::api::AccessScope {
                        stages: StageMask::ALL_COMMANDS,
                        accesses: AccessMask::ALL,
                    },
                }],
            });
            let plan = unsafe { SubmissionPlan::from_compiled(id, std::sync::Arc::from(steps)) };
            plan.validate_for_execution(device)?;
            plan
        } else {
            execution.submission(device, id, lane)?
        };
        let producer = DeferredPlan::capture_namespace(
            device,
            id,
            execution.clone(),
            submission.clone(),
            declaration.producer,
            None,
        )?;
        Ok(Self {
            producer,
            producer_composition: declaration.producer_composition,
            receiver: declaration.receiver,
            terminal: declaration.terminal,
            submission,
            execution,
            id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validate_frames(
        &self,
        device: &Device<B>,
        producer: &DeferredFrame<B>,
        overlay: Option<&super::capture_producer::DeferredCaptureOverlay<'_, B>>,
        receiver: &crate::residency::ForwardFrame<B>,
        receiver_oit: Option<&crate::graph::ForwardOitInput<'_, B>>,
        image: Option<&fluxel_rhi::api::SurfaceImage<B>>,
        composite: Option<&crate::renderer::SurfaceCompositeOwner<B>>,
    ) -> Result<()> {
        super::validation::validate_frame_inputs(device, producer)?;
        self.producer.validate_frame_facts(producer)?;
        self.receiver.validate_frame(device.id(), receiver)?;
        self.receiver
            .validate_oit_input(device.id(), receiver, receiver_oit)?;
        if device.id() != self.id.device
            || producer.extent != receiver.extent
            || producer.deformation.is_some()
        {
            return Err(Error::InvalidState);
        }
        super::forward_capture_owners::validate_owners(
            producer,
            overlay,
            receiver,
            receiver_oit,
            self.receiver.scene_refraction(),
        )?;
        match (&self.terminal, image, composite) {
            (ForwardReceiverTerminal::Offscreen { .. }, None, None) => {}
            (ForwardReceiverTerminal::HdrReadback(_), None, Some(composite)) => {
                let output = receiver.frame.output.as_ref().ok_or(Error::InvalidState)?;
                let source = receiver_oit.map_or(receiver.frame.target_instance, |input| {
                    input.output.instance
                });
                if composite.target_instance() != source || output.instance == source {
                    return Err(Error::InvalidState);
                }
                composite.validate_offscreen(device, receiver.extent, output.desc.format)?;
            }
            (
                ForwardReceiverTerminal::Surface { extent, format, .. },
                Some(image),
                Some(composite),
            ) if image.is_current()
                && image.device() == device.id()
                && image.extent() == *extent
                && image.format() == *format =>
            {
                let instance = receiver_oit.map_or(receiver.frame.target_instance, |input| {
                    input.output.instance
                });
                if composite.target_instance() != instance {
                    return Err(Error::InvalidState);
                }
                composite.validate_for(device, image)?;
            }
            _ => return Err(Error::InvalidState),
        }
        self.bindings(producer, overlay, receiver, receiver_oit, image)
            .map(|_| ())
    }

    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        &self.submission
    }

    pub(crate) fn steps(&self) -> impl Iterator<Item = (StepId, GraphPass)> + '_ {
        let id = self.id;
        self.execution
            .logical()
            .passes()
            .iter()
            .copied()
            .enumerate()
            .map(move |(index, pass)| {
                (
                    StepId {
                        plan: id,
                        index: u32::try_from(index).expect("graph pass index"),
                    },
                    pass,
                )
            })
    }

    pub(crate) fn present_step(&self) -> Result<StepId> {
        match self.terminal {
            ForwardReceiverTerminal::Surface { .. } => Ok(StepId {
                plan: self.id,
                index: u32::try_from(self.execution.logical().passes().len())
                    .map_err(|_| Error::InvalidRange)?,
            }),
            _ => Err(Error::InvalidState),
        }
    }

    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        if matches!(self.terminal, ForwardReceiverTerminal::Surface { .. }) {
            return Err(Error::InvalidState);
        }
        self.producer.capture_completion_all(outcome, None)
    }

    pub(crate) fn surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        self.producer
            .capture_completion_all(outcome, Some(self.present_step()?))
    }
}

fn namespace_passes(declaration: &DeferredDeclaration) -> Vec<GraphPass> {
    let mut passes = vec![declaration.upload_pass];
    passes.extend(declaration.mip_generation.iter().map(|item| item.pass));
    if let Some(pass) = declaration.deformation_pass {
        passes.push(pass);
    }
    passes.push(declaration.geometry_pass);
    if let Some(pass) = declaration.resolve_pass {
        passes.push(pass);
    }
    passes.push(declaration.lighting_pass);
    passes
}
