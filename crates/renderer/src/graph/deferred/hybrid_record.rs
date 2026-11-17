//! Existing Deferred prefix recording against a caller-supplied combined Graph binding set.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(super) fn record_upload_with_bindings(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_upload_with_bindings(
            encoder,
            super::record_view::DeferredRecordView::offscreen(frame),
            bindings,
        )
    }

    pub(super) fn record_mip_with_bindings(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
        generation: &super::super::mip_generation::MipGenerationPass,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_mip_generation_with_bindings(
            encoder,
            super::record_view::DeferredRecordView::offscreen(frame),
            generation,
            bindings,
        )
    }

    pub(super) fn record_geometry_with_bindings(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.record_body_geometry_with_bindings(
            encoder,
            super::record_view::DeferredRecordView::offscreen(frame),
            bindings,
        )
    }
}
