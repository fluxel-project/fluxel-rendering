//! Step projections for the Deferred execution plan.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn mip_generation_steps(
        &self,
    ) -> impl Iterator<Item = (StepId, &super::super::mip_generation::MipGenerationPass)> {
        self.mip_generation.iter().map(|generation| {
            (
                StepId {
                    plan: self.id,
                    index: u32::try_from(generation.pass.1).expect("graph pass index fits StepId"),
                },
                generation,
            )
        })
    }

    pub(crate) fn resolve_step(&self) -> Option<StepId> {
        self.resolve_pass.map(|pass| StepId {
            plan: self.id,
            index: u32::try_from(pass.1).expect("graph pass index fits StepId"),
        })
    }
}
