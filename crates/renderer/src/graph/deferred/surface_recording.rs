//! Surface adapters for Deferred passes shared with offscreen recording.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn surface_record_deformation(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_deformation(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
        )
    }

    pub(crate) fn surface_record_mip_generation(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        generation: &super::super::mip_generation::MipGenerationPass,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_mip_generation(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
            generation,
        )
    }

    pub(crate) fn surface_record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_upload(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
        )
    }

    pub(crate) fn surface_record_geometry(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.record_body_geometry(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
        )
    }
}
