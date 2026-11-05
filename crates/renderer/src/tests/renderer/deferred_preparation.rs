//! Test-only access to the private preparation boundary for owner replacement.

use super::*;

impl<B> Renderer<B>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    pub(in crate::renderer) async fn prepare_deferred_surface_test_frame(
        &mut self,
        input: &crate::ForwardDrawInput,
    ) -> Result<crate::residency::DeferredSurfaceFrame<B>, RendererError> {
        self.prepare_deferred_surface_frame(input, TextureFormat::Rgba8Unorm)
            .await
    }
    pub(in crate::renderer) async fn prepare_deferred_test_frame(
        &mut self,
        input: &crate::ForwardDrawInput,
    ) -> Result<crate::residency::DeferredFrame<B>, RendererError> {
        self.prepare_deferred_frame(input).await
    }

    pub(in crate::renderer) async fn prepare_deformed_test_frame(
        &mut self,
        input: &crate::ForwardDrawInput,
        deformation: &crate::PositionDeformation,
    ) -> Result<crate::residency::DeferredFrame<B>, RendererError> {
        let producer = self.plan_deformation(input, deformation)?;
        let mut frame = self.prepare_deferred_frame(input).await?;
        frame.deformation = Some(self.realize_deformation(producer).await?);
        Ok(frame)
    }

    pub(in crate::renderer) async fn prepare_deformed_surface_test_frame(
        &mut self,
        input: &crate::ForwardDrawInput,
        deformation: &crate::PositionDeformation,
    ) -> Result<crate::residency::DeferredSurfaceFrame<B>, RendererError> {
        let producer = self.plan_deformation(input, deformation)?;
        let mut frame = self
            .prepare_deferred_surface_frame(input, TextureFormat::Rgba8Unorm)
            .await?;
        frame.deformation = Some(self.realize_deformation(producer).await?);
        Ok(frame)
    }
}
