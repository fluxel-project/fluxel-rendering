//! Actual typed producer read declarations for Deferred capture.

use super::*;
use crate::{
    FeatureHandle, ForwardDrawInput, ForwardDrawSource, ForwardPhaseContext, PipelineError,
    RenderFeature, graph::ForwardOitReadContext, pipeline::read_all,
};

#[derive(Default)]
pub(super) struct CaptureDrawGroups {
    meshes: Vec<usize>,
    feature: Vec<usize>,
}

impl CaptureDrawGroups {
    pub(super) fn from_draws(
        draws: &[ForwardDrawInput],
        feature: Option<FeatureHandle>,
    ) -> Result<Self, PipelineError> {
        let mut groups = Self::default();
        if let Some(handle) = feature {
            for (index, input) in draws.iter().enumerate() {
                match input.source() {
                    ForwardDrawSource::Scene(_) => groups.meshes.push(index),
                    ForwardDrawSource::Feature { feature, .. } if feature == handle => {
                        groups.feature.push(index)
                    }
                    _ => return Err(PipelineError::InvalidDrawOrder),
                }
            }
        }
        Ok(groups)
    }
}

pub(super) trait CaptureProducerReads<B: ResourceBackend> {
    fn feature_handle(&self) -> Option<FeatureHandle> {
        None
    }

    fn deferred(
        &self,
        context: &mut crate::DeferredPhaseContext<'_, '_>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError>;

    fn sorted(
        &self,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
        _: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        context.sorted(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_selected_depth()?;
            pass.read_write_main_color()
        })
    }

    fn overlay(
        &self,
        context: &mut ForwardOitReadContext<'_, '_>,
        _: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        context.all_reads()
    }
}

pub(super) struct SceneCaptureProducer<'a, 'frame, P> {
    pub(super) pipeline: &'a P,
    pub(super) selected: &'a crate::PreparedFrame<'frame, crate::ForwardResources>,
}

impl<B: ResourceBackend, P: FramePipeline<crate::ForwardResources>> CaptureProducerReads<B>
    for SceneCaptureProducer<'_, '_, P>
{
    fn deferred(
        &self,
        context: &mut crate::DeferredPhaseContext<'_, '_>,
        _: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        self.pipeline.contribute_deferred(self.selected, context)
    }
}

pub(super) struct FeatureCaptureProducer<'a, 'frame, P, F: RenderFeature> {
    pub(super) pipeline: &'a P,
    pub(super) selected: &'a crate::PreparedFrame<'frame, crate::ForwardResources>,
    pub(super) feature: &'a F,
    pub(super) handle: FeatureHandle,
    pub(super) prepared: &'a F::Prepared,
}

impl<B: ResourceBackend, P: FramePipeline<crate::ForwardResources>, F: RenderFeature>
    CaptureProducerReads<B> for FeatureCaptureProducer<'_, '_, P, F>
{
    fn feature_handle(&self) -> Option<FeatureHandle> {
        Some(self.handle)
    }

    fn deferred(
        &self,
        context: &mut crate::DeferredPhaseContext<'_, '_>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        context.geometry(|pass| {
            pass.depends_on_uploads()?;
            pass.scope(&groups.meshes, read_all)?;
            pass.scope(&groups.feature, |scope| {
                self.feature.contribute(self.prepared, scope)
            })?;
            pass.write_gbuffer()
        })?;
        context.lighting(|pass| {
            pass.read_gbuffer()?;
            pass.write_main_color()
        })
    }

    fn sorted(
        &self,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        let mut composed = crate::ForwardCompositionContext::new(
            context,
            self.feature,
            self.prepared,
            &groups.meshes,
            &groups.feature,
        );
        self.pipeline
            .contribute_composed(self.selected, &mut composed)
    }

    fn overlay(
        &self,
        context: &mut ForwardOitReadContext<'_, '_>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        context.scope(&groups.meshes, read_all)?;
        context.scope(&groups.feature, |scope| {
            self.feature.contribute(self.prepared, scope)
        })
    }
}
