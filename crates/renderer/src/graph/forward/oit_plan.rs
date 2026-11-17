//! OIT metadata belonging to the same compiled Forward Graph and submission.

use super::*;
use crate::graph::{oit::DualDepthPeelGraph, oit_record::OitRecordSlots};

#[derive(Clone, Copy)]
pub(super) struct OitSurfaceTarget {
    pub(super) slot: TextureSlot,
    pub(super) extent: [u32; 2],
    pub(super) format: TextureFormat,
}

pub(super) struct ForwardOitPlan {
    pub(super) product: DualDepthPeelGraph,
    pub(super) slots: OitRecordSlots,
    pub(super) opaque_count: usize,
    pub(super) peel_count: usize,
    pub(super) rounds: NonZeroU32,
    pub(super) fallback: Option<GraphPass>,
    pub(super) surface: Option<OitSurfaceTarget>,
}

impl ForwardOitPlan {
    pub(super) fn passes(&self) -> Vec<GraphPass> {
        let passes = &self.product.passes;
        let mut result = vec![
            passes.initial_clear[0].pass,
            passes.initial_clear[1].pass,
            passes.initialize_back_accumulator,
            passes.initial_depth,
        ];
        for iteration in &passes.iterations {
            result.extend([
                iteration.clear.pass,
                iteration.peel,
                iteration.back_accumulate,
            ]);
        }
        result.push(passes.final_composite);
        result
    }
}

impl<B: ResourceBackend> ForwardPlan<B> {
    pub(crate) fn declare_with_oit_contribution<C>(
        device: &Device<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        contribute: C,
    ) -> std::result::Result<
        super::oit_declaration::OffscreenForwardOitDeclaration,
        ForwardPlanBuildError,
    >
    where
        C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
    {
        super::oit_declaration::declare_with_contribution(device, frame, input, contribute)
    }

    pub(crate) fn declare_surface_oit_contribution<C>(
        device: &Device<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        image: &SurfaceImage<B>,
        contribute: C,
    ) -> std::result::Result<
        super::oit_declaration::OffscreenForwardOitDeclaration,
        ForwardPlanBuildError,
    >
    where
        C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
    {
        super::oit_declaration::declare_surface_with_contribution(
            device, frame, input, image, contribute,
        )
    }

    pub(crate) fn compile_declared_oit(
        device: &Device<B>,
        id: PlanId,
        declaration: super::oit_declaration::OffscreenForwardOitDeclaration,
    ) -> std::result::Result<Self, ForwardPlanBuildError> {
        Self::compile_declared_inner(device, id, declaration.base, Some(declaration.oit))
    }

    pub(crate) fn validate_frame_with_oit(
        &self,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()> {
        let product = self.oit.as_ref().ok_or(Error::InvalidState)?;
        if product.opaque_count != input.opaque_count
            || product.peel_count != input.peel_count
            || product.rounds != input.rounds
            || input
                .opaque_count
                .checked_add(input.peel_count)
                .is_none_or(|end| end > frame.draws.len())
            || input.owners.extent != frame.extent
            || input.output.desc.extent != [frame.extent[0], frame.extent[1], 1]
            || input.output.device != self.id.device
            || input.output.desc.format != frame.frame.target_desc.format
            || input.output.view.device() != self.id.device
            || input.output.view_desc.format != input.output.desc.format
            || input.output.view_desc.dimension != fluxel_rhi::api::TextureViewDimension::D2
            || input.output.view_desc.aspects != TextureAspects::COLOR
            || input.output.view_desc.mips != (0..1)
            || input.output.view_desc.layers != (0..1)
        {
            return Err(Error::InvalidState);
        }
        match (product.surface, image) {
            (None, None) => {}
            (Some(surface), Some(image)) => {
                if image.device() != self.id.device {
                    return Err(Error::WrongDevice);
                }
                if !image.is_current() {
                    return Err(Error::SurfaceOutdated);
                }
                if image.extent() != surface.extent || image.format() != surface.format {
                    return Err(Error::InvalidRange);
                }
            }
            _ => return Err(Error::InvalidState),
        }
        input.owners.validate(self.id.device)?;
        self.validate_frame_base(frame)?;
        self.bindings_with_oit(frame, input, image).map(|_| ())
    }

    /// These identities come from the compiled Graph order. The renderer
    /// records one supplied pass per step, with no independent OIT scheduler.
    pub(crate) fn oit_steps(&self) -> impl Iterator<Item = (StepId, GraphPass)> + '_ {
        let passes = self
            .oit
            .as_ref()
            .map(|product| product.passes())
            .unwrap_or_default();
        self.execution
            .logical()
            .passes()
            .iter()
            .copied()
            .enumerate()
            .filter(move |(_, pass)| passes.contains(pass))
            .map(|(index, pass)| {
                (
                    StepId {
                        plan: self.id,
                        index: u32::try_from(index).expect("compiled pass index fits u32"),
                    },
                    pass,
                )
            })
    }

    pub(crate) fn oit_fallback_step(&self) -> Option<StepId> {
        let pass = self.oit.as_ref()?.fallback?;
        self.execution
            .logical()
            .passes()
            .iter()
            .position(|candidate| *candidate == pass)
            .map(|index| StepId {
                plan: self.id,
                index: u32::try_from(index).expect("compiled pass index fits u32"),
            })
    }

    pub(crate) fn oit_surface_step(&self) -> Result<StepId> {
        if self.oit.as_ref().and_then(|oit| oit.surface).is_none() {
            return Err(Error::InvalidState);
        }
        Ok(self.readback_step())
    }

    pub(crate) fn oit_present_step(&self) -> Result<StepId> {
        self.oit_surface_step()?;
        Ok(StepId {
            plan: self.id,
            index: u32::try_from(self.execution.logical().passes().len())
                .map_err(|_| Error::InvalidRange)?,
        })
    }

    pub(crate) fn surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let present = self.oit_present_step()?;
        let mut expected: Vec<_> = (0..self.execution.logical().passes().len())
            .map(|index| StepId {
                plan: self.id,
                index: u32::try_from(index).expect("compiled pass index fits u32"),
            })
            .collect();
        expected.push(present);
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(expected.as_slice())
            || !facts
                .presents
                .is_some_and(|p| p.len() == 1 && p[0].accepted && p[0].step == Some(present))
        {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|points| {
                points.iter().find(|point| {
                    point.point.device() == self.id.device
                        && point
                            .covers
                            .lane_frontiers
                            .iter()
                            .any(|&(lane, step)| lane == self.lane && step == present)
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }
}
