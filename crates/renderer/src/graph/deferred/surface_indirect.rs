//! Surface recording adapters for Deferred indirect producers.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn surface_record_indirect_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        draw: usize,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_indirect_upload(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
            draw,
        )
    }

    pub(crate) fn surface_record_indirect_compute(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        draw: usize,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_indirect_compute(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
            draw,
        )
    }
}
