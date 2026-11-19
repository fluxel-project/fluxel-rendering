//! Indirect producer step lookup for a Deferred plan.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    /// Returns the upload and compute steps for each retained indirect record.
    pub(crate) fn indirect_steps(&self) -> impl Iterator<Item = (usize, StepId, StepId)> + '_ {
        self.indirect.iter().map(|(draw, graph)| {
            (
                *draw,
                StepId {
                    plan: self.id,
                    index: u32::try_from(graph.upload_pass.1)
                        .expect("graph pass index fits StepId"),
                },
                StepId {
                    plan: self.id,
                    index: u32::try_from(graph.compute_pass.1)
                        .expect("graph pass index fits StepId"),
                },
            )
        })
    }
}
