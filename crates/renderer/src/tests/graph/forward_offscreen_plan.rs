use super::*;

impl<B: ResourceBackend> ForwardPlan<B> {
    /// Builds the handle-free Graph facts from the admitted owner, then checks
    /// the owner separately while binding the resulting plan.
    pub(crate) fn compile_with<F>(
        device: &Device<B>,
        id: PlanId,
        frame: &ForwardFrame<B>,
        expected: FrameTopology,
        contribute: F,
    ) -> std::result::Result<Self, ForwardPlanBuildError>
    where
        F: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
    {
        let declaration = Self::declare_with(device, frame, expected, contribute)?;
        let plan = Self::compile_declared(device, id, declaration)?;
        plan.validate_frame(frame)?;
        Ok(plan)
    }
}
