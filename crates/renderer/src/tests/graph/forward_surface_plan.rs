use super::*;

impl<B: ResourceBackend> SurfaceForwardPlan<B> {
    pub(crate) fn compile_with<F>(
        device: &Device<B>,
        id: PlanId,
        frame: &SurfaceForwardFrame<B>,
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
