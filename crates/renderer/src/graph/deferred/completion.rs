//! Completion evidence for Deferred plans.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let mut expected = vec![self.upload_step()];
        expected.extend(self.mip_generation_steps().map(|(step, _)| step));
        for (_, upload, compute) in self.indirect_steps() {
            expected.extend([upload, compute]);
        }
        if let Some(step) = self.deformation_step() {
            expected.push(step);
        }
        expected.push(self.geometry_step());
        if let Some(step) = self.resolve_step() {
            expected.push(step);
        }
        expected.push(self.lighting_step());
        expected.extend(self.composition_passes.iter().map(|pass| StepId {
            plan: self.id,
            index: u32::try_from(pass.1).expect("graph pass index fits StepId"),
        }));
        if self.composite_pass.is_some() {
            expected.push(self.composite_step());
        }
        if matches!(self.output, DeferredOutputPlan::Offscreen { .. }) {
            expected.push(self.readback_step());
        }
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(&expected) {
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
                            .any(|&(lane, step)| lane == self.lane && step == self.readback_step())
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }

    pub(crate) fn surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let present = self.present_step()?;
        let mut expected = vec![self.upload_step()];
        expected.extend(self.mip_generation_steps().map(|(step, _)| step));
        for (_, upload, compute) in self.indirect_steps() {
            expected.extend([upload, compute]);
        }
        if let Some(step) = self.deformation_step() {
            expected.push(step);
        }
        expected.push(self.geometry_step());
        if let Some(step) = self.resolve_step() {
            expected.push(step);
        }
        expected.push(self.lighting_step());
        expected.extend(self.composition_passes.iter().map(|pass| StepId {
            plan: self.id,
            index: u32::try_from(pass.1).expect("graph pass index fits StepId"),
        }));
        expected.extend([self.composite_step(), present]);
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(&expected)
            || !facts.presents.is_some_and(|items| {
                items.len() == 1 && items[0].accepted && items[0].step == Some(present)
            })
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
