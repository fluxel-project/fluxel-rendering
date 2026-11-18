//! Compiled single-Graph Forward-to-Deferred capture plan.

use super::reverse_capture_declaration::ForwardDeferredCaptureDeclaration;
use super::*;

pub(crate) struct ForwardDeferredCapturePlan<B: ResourceBackend> {
    pub(super) producer: crate::graph::forward::ForwardCaptureReceiver,
    pub(super) receiver: DeferredPlan<B>,
    pub(super) submission: SubmissionPlan<B>,
    pub(super) execution: CompiledExecution,
    pub(super) id: PlanId,
    pub(super) surface: Option<([u32; 2], TextureFormat, StepId)>,
}

impl<B: ResourceBackend> ForwardDeferredCapturePlan<B> {
    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: ForwardDeferredCaptureDeclaration,
    ) -> Result<Self> {
        if id.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let execution = CompiledExecution::compile(&declaration.receiver.graph)?;
        let mut expected = Vec::new();
        declaration.producer.append_passes(&mut expected);
        expected.extend(namespace_passes(&declaration.receiver));
        if let Some(pass) = declaration.receiver.composite_pass {
            expected.push(pass);
        }
        if let DeferredOutputDeclaration::Offscreen { readback_pass } = declaration.receiver.output
        {
            expected.push(readback_pass);
        }
        if execution.logical().passes() != expected {
            return Err(Error::InvalidState);
        }
        let (submission, surface) = match declaration.surface {
            None => (
                execution.submission(device, id, declaration.receiver.lane)?,
                None,
            ),
            Some((extent, format)) => {
                let index = u32::try_from(expected.len()).map_err(|_| Error::InvalidRange)?;
                let present = StepId { plan: id, index };
                let mut steps = execution.steps(device, id, declaration.receiver.lane)?;
                steps.push(fluxel_rhi::api::StepDesc {
                    lane: declaration.receiver.lane,
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
                let submission =
                    unsafe { SubmissionPlan::from_compiled(id, std::sync::Arc::from(steps)) };
                submission.validate_for_execution(device)?;
                (submission, Some((extent, format, present)))
            }
        };
        let receiver = DeferredPlan::capture_namespace(
            device,
            id,
            execution.clone(),
            submission.clone(),
            declaration.receiver,
            surface.map(|x| x.2),
        )?;
        Ok(Self {
            producer: declaration.producer,
            receiver,
            submission,
            execution,
            id,
            surface,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validate_frames(
        &self,
        device: &Device<B>,
        producer: &crate::residency::ForwardFrame<B>,
        producer_oit: Option<&crate::graph::ForwardOitInput<'_, B>>,
        receiver: &DeferredFrame<B>,
        image: Option<&fluxel_rhi::api::SurfaceImage<B>>,
        composite: Option<&crate::renderer::SurfaceCompositeOwner<B>>,
    ) -> Result<()> {
        super::validation::validate_frame_inputs(device, receiver)?;
        self.producer.validate_frame(device.id(), producer)?;
        self.producer
            .validate_oit_input(device.id(), producer, producer_oit)?;
        self.receiver.validate_frame_facts(receiver)?;
        if device.id() != self.id.device
            || producer.extent != receiver.extent
            || receiver.deformation.is_some()
        {
            return Err(Error::InvalidState);
        }
        super::reverse_capture_owners::validate_owners(producer, producer_oit, receiver)?;
        match (self.surface, image, composite) {
            (None, None, None) => {}
            (Some((extent, format, _)), Some(image), Some(composite))
                if image.is_current()
                    && image.device() == device.id()
                    && image.extent() == extent
                    && image.format() == format
                    && composite.target_instance() == receiver.scene_color.instance =>
            {
                composite.validate_for(device, image)?
            }
            _ => return Err(Error::InvalidState),
        }
        self.bindings(producer, producer_oit, receiver, image)
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
            .map(move |(i, p)| {
                (
                    StepId {
                        plan: id,
                        index: u32::try_from(i).expect("graph pass index"),
                    },
                    p,
                )
            })
    }
    pub(crate) fn present_step(&self) -> Result<StepId> {
        self.surface.map(|x| x.2).ok_or(Error::InvalidState)
    }
    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        if self.surface.is_some() {
            return Err(Error::InvalidState);
        }
        self.receiver.capture_completion_all(outcome, None)
    }
    pub(crate) fn surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        self.receiver
            .capture_completion_all(outcome, Some(self.present_step()?))
    }
}

fn namespace_passes(declaration: &DeferredDeclaration) -> Vec<GraphPass> {
    let mut passes = vec![declaration.upload_pass];
    passes.extend(declaration.mip_generation.iter().map(|x| x.pass));
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
