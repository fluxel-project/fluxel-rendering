//! Compilation, validation, and completion for one Deferred capture Graph.

use super::capture_declaration::DeferredCaptureDeclaration;
use super::*;

pub(crate) struct DeferredCapturePlan<B: ResourceBackend> {
    pub(super) producer: DeferredPlan<B>,
    pub(super) receiver: DeferredPlan<B>,
    pub(super) producer_composition:
        Option<super::capture_producer::DeferredCaptureProducerComposition>,
    pub(super) submission: SubmissionPlan<B>,
    pub(super) execution: CompiledExecution,
    pub(super) id: PlanId,
    surface: Option<([u32; 2], TextureFormat, StepId)>,
}

impl<B: ResourceBackend> DeferredCapturePlan<B> {
    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: DeferredCaptureDeclaration,
    ) -> Result<Self> {
        if id.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let execution = CompiledExecution::compile(&declaration.receiver.graph)?;
        let mut expected = namespace_passes(&declaration.producer);
        if let Some(composition) = declaration.producer_composition.as_ref() {
            composition.append_passes(&mut expected);
        }
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
        let producer = DeferredPlan::capture_namespace(
            device,
            id,
            execution.clone(),
            submission.clone(),
            declaration.producer,
            None,
        )?;
        let receiver = DeferredPlan::capture_namespace(
            device,
            id,
            execution.clone(),
            submission.clone(),
            declaration.receiver,
            surface.map(|(_, _, present)| present),
        )?;
        Ok(Self {
            producer,
            receiver,
            producer_composition: declaration.producer_composition,
            submission,
            execution,
            id,
            surface,
        })
    }

    pub(crate) fn validate_frames(
        &self,
        device: &Device<B>,
        producer: &DeferredFrame<B>,
        producer_overlay: Option<&super::capture_producer::DeferredCaptureOverlay<'_, B>>,
        receiver: &DeferredFrame<B>,
        image: Option<&fluxel_rhi::api::SurfaceImage<B>>,
        composite: Option<&crate::renderer::SurfaceCompositeOwner<B>>,
    ) -> Result<()> {
        super::validation::validate_frame_inputs(device, producer)?;
        super::validation::validate_frame_inputs(device, receiver)?;
        if device.id() != self.id.device
            || producer.extent != receiver.extent
            || producer.deformation.is_some()
            || receiver.deformation.is_some()
        {
            return Err(Error::InvalidState);
        }
        self.producer.validate_frame_facts(producer)?;
        self.receiver.validate_frame_facts(receiver)?;
        super::capture_owner_validation::validate_capture_owners(
            producer,
            producer_overlay,
            receiver,
        )?;
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
        match image {
            Some(image) => self
                .surface_bindings(producer, producer_overlay, receiver, image)
                .map(|_| ()),
            None => self
                .bindings(producer, producer_overlay, receiver)
                .map(|_| ()),
        }
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
    passes.extend([declaration.geometry_pass, declaration.lighting_pass]);
    passes
}
