//! Borrowed actual owners shared by the two Deferred recording consumers.

use super::*;
use crate::residency::{
    DeferredDeformation, ForwardResidentBuffers, ForwardResidentUpload, ForwardTexture,
    ForwardUniformUpload,
};

pub(super) enum RecordSource<'a, B: ResourceBackend> {
    Offscreen(&'a DeferredFrame<B>),
    Surface {
        frame: &'a DeferredSurfaceFrame<B>,
        image: Option<&'a fluxel_rhi::api::SurfaceImage<B>>,
    },
}

pub(super) struct DeferredRecordView<'a, B: ResourceBackend> {
    pub(super) source: RecordSource<'a, B>,
    pub(super) frame: UniformFacts<'a>,
    pub(super) resident: &'a ForwardResidentBuffers<B>,
    pub(super) resident_upload: &'a ForwardResidentUpload,
    pub(super) uniform_upload: &'a ForwardUniformUpload,
    pub(super) environment: Option<&'a crate::residency::EnvironmentFrameOwner<B>>,
    pub(super) specular: Option<&'a crate::residency::SpecularEnvironmentFrameOwner<B>>,
    pub(super) deformation: Option<&'a DeferredDeformation<B>>,
    pub(super) gbuffer: &'a [ForwardTexture<B>; 3],
    pub(super) effects: Option<&'a ForwardTexture<B>>,
    pub(super) draws: &'a [DeferredDraw<B>],
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
            environment: frame.environment.as_ref(),
            specular: frame.specular.as_ref(),
            deformation: frame.deformation.as_ref(),
            gbuffer: &frame.gbuffer,
            effects: frame.effects.as_ref(),
            draws: &frame.draws,
            extent: frame.extent,
        }
    }

    pub(super) fn surface(frame: &'a DeferredSurfaceFrame<B>) -> Self {
        Self {
            source: RecordSource::Surface { frame, image: None },
            frame: UniformFacts {
                upload_desc: &frame.frame.upload_desc,
                upload_incoming: frame.frame.upload_incoming,
                uniform_desc: &frame.frame.uniform_desc,
                uniform_incoming: frame.frame.uniform_incoming,
            },
            resident: &frame.resident,
            resident_upload: &frame.resident_upload,
            uniform_upload: &frame.uniform_upload,
            environment: frame.environment.as_ref(),
            specular: frame.specular.as_ref(),
            deformation: frame.deformation.as_ref(),
            gbuffer: &frame.gbuffer,
            effects: frame.effects.as_ref(),
            draws: &frame.draws,
            extent: frame.extent,
        }
    }

    pub(super) fn surface_with_image(
        frame: &'a DeferredSurfaceFrame<B>,
        image: &'a fluxel_rhi::api::SurfaceImage<B>,
    ) -> Self {
        let mut view = Self::surface(frame);
        view.source = RecordSource::Surface {
            frame,
            image: Some(image),
        };
        view
    }

    pub(super) fn draw_graph_data(&self) -> Result<Vec<super::phase::DeferredGraphDraw>> {
        if self.draws.is_empty() || (self.deformation.is_some() && self.draws.len() != 1) {
            return Err(Error::InvalidRange);
        }
        if self.draws.iter().any(|draw| {
            draw.textures.is_empty()
                || draw.capture_textures.len() != draw.textures.len()
                || draw
                    .textures
                    .iter()
                    .any(|texture| *texture >= self.resident.textures.len())
        }) {
            return Err(Error::InvalidRange);
        }
        match self.source {
            RecordSource::Offscreen(frame) => frame
                .draws
                .iter()
                .map(|draw| {
                    Ok(super::phase::DeferredGraphDraw {
                        streams: draw.vertex_streams.clone(),
                        accesses: draw_accesses(frame, draw)?,
                        tint: draw.tint_range.clone(),
                        transform: draw.transform_range.clone(),
                        textures: draw.textures.clone(),
                    })
                })
                .collect(),
            RecordSource::Surface { frame, .. } => frame
                .draws
                .iter()
                .map(|draw| {
                    Ok(super::phase::DeferredGraphDraw {
                        streams: draw.vertex_streams.clone(),
                        accesses: super::surface_validate::surface_draw_accesses(frame, draw)?,
                        tint: draw.tint_range.clone(),
                        transform: draw.transform_range.clone(),
                        textures: draw.textures.clone(),
                    })
                })
                .collect(),
        }
    }
}

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(super) fn record_bindings<'a>(
        &'a self,
        view: &DeferredRecordView<'a, B>,
    ) -> Result<FrameBindings<'a, B>> {
        if let Some(specular) = view.specular {
            // Recheck the concrete owner immediately before converting Graph
            // accesses into backend commands; declarations carry no identity.
            specular.validate_for_sampling(self.id.device)?;
        }
        match view.source {
            RecordSource::Offscreen(frame) => self.bindings(frame),
            RecordSource::Surface { frame, image } => {
                self.surface_bindings(frame, image.ok_or(fluxel_rhi::api::Error::InvalidState)?)
            }
        }
    }
}
