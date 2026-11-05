//! Borrowed actual owners shared by the two Deferred recording consumers.

use super::*;
use crate::residency::{
    DeferredDeformation, ForwardResidentBuffers, ForwardResidentUpload, ForwardTexture,
    ForwardUniformUpload,
};

pub(super) enum RecordSource<'a, B: ResourceBackend> {
    Offscreen(&'a DeferredFrame<B>),
    Surface(&'a DeferredSurfaceFrame<B>),
}

pub(super) struct DeferredRecordView<'a, B: ResourceBackend> {
    pub(super) source: RecordSource<'a, B>,
    pub(super) frame: UniformFacts<'a>,
    pub(super) resident: &'a ForwardResidentBuffers<B>,
    pub(super) resident_upload: &'a ForwardResidentUpload,
    pub(super) uniform_upload: &'a ForwardUniformUpload,
    pub(super) deformation: Option<&'a DeferredDeformation<B>>,
    pub(super) gbuffer: &'a [ForwardTexture<B>; 3],
    pub(super) draw: &'a DeferredDraw<B>,
    pub(super) extent: [u32; 2],
}

pub(super) struct UniformFacts<'a> {
    pub(super) upload_desc: &'a fluxel_rhi::api::BufferDesc,
    pub(super) upload_incoming: BufferUses,
    pub(super) uniform_desc: &'a fluxel_rhi::api::BufferDesc,
    pub(super) uniform_incoming: BufferUses,
}

impl<'a, B: ResourceBackend> DeferredRecordView<'a, B> {
    pub(super) fn offscreen(frame: &'a DeferredFrame<B>) -> Self {
        Self {
            source: RecordSource::Offscreen(frame),
            frame: UniformFacts {
                upload_desc: &frame.frame.upload_desc,
                upload_incoming: frame.frame.upload_incoming,
                uniform_desc: &frame.frame.uniform_desc,
                uniform_incoming: frame.frame.uniform_incoming,
            },
            resident: &frame.resident,
            resident_upload: &frame.resident_upload,
            uniform_upload: &frame.uniform_upload,
            deformation: frame.deformation.as_ref(),
            gbuffer: &frame.gbuffer,
            draw: &frame.draw,
            extent: frame.extent,
        }
    }

    pub(super) fn surface(frame: &'a DeferredSurfaceFrame<B>) -> Self {
        Self {
            source: RecordSource::Surface(frame),
            frame: UniformFacts {
                upload_desc: &frame.frame.upload_desc,
                upload_incoming: frame.frame.upload_incoming,
                uniform_desc: &frame.frame.uniform_desc,
                uniform_incoming: frame.frame.uniform_incoming,
            },
            resident: &frame.resident,
            resident_upload: &frame.resident_upload,
            uniform_upload: &frame.uniform_upload,
            deformation: frame.deformation.as_ref(),
            gbuffer: &frame.gbuffer,
            draw: &frame.draw,
            extent: frame.extent,
        }
    }

    pub(super) fn draw_accesses(&self) -> Result<Vec<Option<Range<u64>>>> {
        match self.source {
            RecordSource::Offscreen(frame) => draw_accesses(frame, &frame.draw),
            RecordSource::Surface(frame) => {
                super::surface_validate::surface_draw_accesses(frame, &frame.draw)
            }
        }
    }
}

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(super) fn record_bindings<'a>(
        &'a self,
        view: &DeferredRecordView<'a, B>,
    ) -> Result<FrameBindings<'a, B>> {
        match view.source {
            RecordSource::Offscreen(frame) => self.bindings(frame),
            RecordSource::Surface(frame) => self.surface_bindings(frame),
        }
    }
}
